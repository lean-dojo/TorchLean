/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

GELU activation for CROWN bound propagation.
GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF.
Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

This file provides Float-specialized implementations for GELU bounds.
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# GELU Operator Bounds

Float-specialized GELU transfer rules for CROWN/IBP-style bound propagation.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

/-!
# GELU Activation (Float-specialized)

GELU is a smooth, non-monotone activation function used in Transformers.
We implement IBP and affine bounds using the tanh approximation.

Key properties:
- GELU(0) = 0
- GELU is approximately linear for large |x|
- Has a small bump near x ≈ -1.5

Because GELU requires specific numerical constants (√(2/π) ≈ 0.7978845608),
we provide Float-specialized implementations.

## Soundness status

The implementations below use:
- the common tanh-based GELU approximation, and
- interval estimates based on endpoint sampling plus a fixed critical-point estimate.

These estimates are executable transfer rules, not unconditional enclosure theorems. Use them under
an explicit transfer-soundness assumption or replace them with a proved relaxation when the final
claim requires unconditional CROWN soundness.

## References

- Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)", 2016:
  https://arxiv.org/abs/1606.08415
- The tanh approximation used here appears in the GELU paper as a fast approximation to the
  Gaussian CDF-based definition.
-/

/-- GELU approximation using tanh: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))) -/
def geluApproxFloat (x : Float) : Float :=
  let sqrt2pi : Float := 0.7978845608  -- √(2/π)
  let c : Float := 0.044715
  let inner := sqrt2pi * (x + c * x * x * x)
  0.5 * x * (1.0 + Float.tanh inner)

/-- GELU derivative approximation -/
def geluDerivApproxFloat (x : Float) : Float :=
  let sqrt2pi : Float := 0.7978845608
  let c : Float := 0.044715
  let inner := sqrt2pi * (x + c * x * x * x)
  let tanh_inner := Float.tanh inner
  let sech2 := 1.0 - tanh_inner * tanh_inner
  let dInner := sqrt2pi * (1.0 + 3.0 * c * x * x)
  -- d/dx [0.5 * x * (1 + tanh(inner))]
  -- = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * dInner
  0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * dInner

/-!
# IBP for GELU

Since GELU is not monotone, we need to be careful about the bounds.
For simplicity, we use sampling at endpoints and critical point.
-/

/-- Approximate min/max of GELU on `[l, u]` (Float version).

This helper is executable and useful for certificate experiments; it is not itself a proof that the
returned interval encloses the mathematical GELU.
-/
def geluIntervalBoundsFloat (l u : Float) : Float × Float :=
  -- Critical point is approximately at x ≈ -1.7
  let criticalPt : Float := -1.7
  let vl := geluApproxFloat l
  let vu := geluApproxFloat u
  let minEndpoints := if vl < vu then vl else vu
  let maxEndpoints := if vl > vu then vl else vu
  -- Check if critical point is in interval
  if l < criticalPt && criticalPt < u then
    let vc := geluApproxFloat criticalPt
    let minVal := if vc < minEndpoints then vc else minEndpoints
    -- GELU minimum at critical point is approximately -0.17
    (minVal, maxEndpoints)
  else
    (minEndpoints, maxEndpoints)

/-- IBP for GELU activation (Float specialized) -/
def ibpGeluFloat {n : Nat} (xB : Box Float (.dim n .scalar)) : Box Float (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (minVal, _) := geluIntervalBoundsFloat l u
        Tensor.scalar minVal)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (_, maxVal) := geluIntervalBoundsFloat l u
        Tensor.scalar maxVal)
    { lo := outLo, hi := outHi }

/-!
# CROWN Affine Bounds for GELU

For CROWN, we need linear upper and lower bounds on GELU.
We use a simple secant/tangent approach:
- Upper bound: secant line if concave, tangent if convex
- Lower bound: tangent if concave, secant if convex
-/

/-- Relaxation parameters for GELU -/
structure GELURelax where
  /-- Slope of the lower affine relaxation candidate. -/
  slope_lower : Float
  /-- Bias of the lower affine relaxation candidate. -/
  bias_lower : Float
  /-- Slope of the upper affine relaxation candidate. -/
  slope_upper : Float
  /-- Bias of the upper affine relaxation candidate. -/
  bias_upper : Float

/-- Compute an affine relaxation candidate for GELU on `[l, u]` (Float version).

This secant-style rule is executable. A theorem that uses it as a verifier step should carry the
corresponding transfer-soundness assumption.
-/
def geluRelaxFloat (l u : Float) : GELURelax :=
  let vl := geluApproxFloat l
  let vu := geluApproxFloat u
  let width := u - l
  -- Secant slope
  let secantSlope := if width > 1e-6 then (vu - vl) / width else geluDerivApproxFloat l
  let secantBias := vl - secantSlope * l
  -- Use the secant for both affine forms; transfer soundness must be supplied by the caller.
  { slope_lower := secantSlope
  , bias_lower := secantBias
  , slope_upper := secantSlope
  , bias_upper := secantBias }

/-- Compute GELU relaxation per element -/
def geluRelaxVectorFloat {n : Nat} (lo hi : Tensor Float (.dim n .scalar)) :
    Tensor GELURelax (.dim n .scalar) :=
  match lo, hi with
  | .dim loVec, .dim hiVec =>
    Tensor.dim (fun i =>
      match loVec i, hiVec i with
      | .scalar l, .scalar u => Tensor.scalar (geluRelaxFloat l u))

/-- CROWN affine propagation through GELU (Float specialized) -/
def crownGeluAffineFloat {inDim outDim : Nat}
    (relax : Tensor GELURelax (.dim outDim .scalar))
    (aff : AffineVec Float inDim outDim) : AffineVec Float inDim outDim :=
  match relax, aff.A, aff.c with
  | .dim r, .dim rows, .dim bias =>
    -- Propagate through the upper affine form selected by `geluRelaxFloat`.
    let A' := Tensor.dim (fun i =>
      match rows i, r i with
      | .dim cols, .scalar rp =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar aij => Tensor.scalar (aij * rp.slope_upper)))
    let c' := Tensor.dim (fun i =>
      match bias i, r i with
      | .scalar ci, .scalar rp =>
        Tensor.scalar (rp.slope_upper * ci + rp.bias_upper))
    { A := A', c := c' }

/-!
# Generic versions using Context

For polymorphic code, we provide versions that work with any Context type
by computing bounds using simpler approximations.
-/

variable {α : Type} [Context α]

/-- ReLU-shaped GELU bound used for generic `Context` scalars.
    For x > 0: GELU(x) ≈ x
    For x < 0: GELU(x) ≈ 0

The theorem using this helper must provide the relevant transfer-soundness hypothesis for the
chosen scalar semantics and interval restrictions. -/
def geluBoundsConservative (l u : α) : α × α :=
  -- Lower proxy: min(0, l). Upper proxy: max(0, u).
  let zeroVal : α := Numbers.zero
  let minBound := if l < zeroVal then l else zeroVal
  let maxBound := if u > zeroVal then u else zeroVal
  (minBound, maxBound)

/-- IBP for GELU activation (generic, conservative) -/
def ibpGelu {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (minVal, _) := geluBoundsConservative l u
        Tensor.scalar minVal)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let (_, maxVal) := geluBoundsConservative l u
        Tensor.scalar maxVal)
    { lo := outLo, hi := outHi }

end NN.MLTheory.CROWN.Operators
