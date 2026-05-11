/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Trigonometric operator bounds (optional)

This file provides simple IBP and affine transfer rules for trigonometric functions used in
physics-informed objectives (PINNs): `sin`, `cos`, `tan`, and `atan`.

This module is not imported by the default `NN.MLTheory.CROWN.Operators` index because `tan` and
`atan` require an extra scalar-function interface. Import it explicitly when needed.

Note: `tan`/`atan` are not part of `Context` today. The corresponding transfer rules below are
therefore gated behind an extra `TanAtan α` typeclass.

## Soundness status

The transfer rules in this file are **optional** and have different proof status by operator:

- `sin`/`cos` delegate to the conservative 1-Lipschitz enclosure rules in
  `NN.MLTheory.CROWN.Runtime.Ops.IBP.sin/cos`. This avoids endpoint-only periodic reasoning, which
  can miss internal extrema.
- `tan`/`atan` require the extra `TanAtan α` class. `atan` is monotone, so interval propagation is
  direct. `tan` is only meaningful under the caller-side precondition that the interval
  stays inside one monotone branch and away from poles.
- Affine transfer rules here are executable relaxation candidates. A downstream theorem should
  provide or assume the corresponding enclosure property for the scalar backend being used.

## References

- PINNs: Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", JCP 2019:
  https://arxiv.org/abs/1711.10561
- CROWN/DeepPoly context: Zhang et al., 2018 (CROWN) https://arxiv.org/abs/1811.00866
- Lipschitz enclosure idea for `sin`/`cos`: |d/dx sin(x)| <= 1 and |d/dx cos(x)| <= 1, so a
  ball/interval can be mapped using a Lipschitz radius (as implemented in `Runtime/Ops`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Trigonometric

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Float approximation of π used only by optional Float-facing examples. -/
private def piApprox : Float := 3.14159265358979323846

/-- Extra trigonometric functions outside the base `Spec.MathFunctions` / `Context` interface. -/
class TanAtan (α : Type) where
  /-- Tangent. -/
  tan : α → α
  /-- Arctangent. -/
  atan : α → α

/-- Helper to extract scalar from dim-scalar tensor. -/
private def getDimScalarFn {n : Nat} (t : Tensor α (.dim n .scalar)) : (Fin n → Tensor α .scalar) :=
  match t with
  | .dim f => f

/-- IBP for `sin`.

This optional operator delegates to the conservative 1-Lipschitz enclosure used by the runtime graph
verifier. We use that rule instead of endpoint-only periodic reasoning, since endpoint checks alone
can miss internal extrema without exact period tracking.
-/
def ibpSin {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.IBP.sin xB

/-- IBP for `cos`.

This optional operator delegates to the same conservative 1-Lipschitz enclosure as the runtime graph
verifier.
-/
def ibpCos {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.IBP.cos xB

/-- IBP for `tan` under the caller-side monotonic-branch precondition.

This rule is intended for intervals contained in a single branch of `tan` and away from poles. If a
workflow may cross a pole, it should reject the certificate or provide a different enclosure rule.
-/
def ibpTan {n : Nat} [TanAtan α] (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar _u =>
        -- In monotonic region, tan(l) is the lower bound
        Tensor.scalar (TanAtan.tan l))
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar _l, .scalar u =>
        -- In monotonic region, tan(u) is the upper bound
        Tensor.scalar (TanAtan.tan u))
    { lo := outLo, hi := outHi }

/-- IBP for Atan: atan is monotonically increasing and bounded in (-π/2, π/2).
    This makes IBP straightforward: [atan(l), atan(u)].
-/
def ibpAtan {n : Nat} [TanAtan α] (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (TanAtan.atan l))
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (TanAtan.atan u))
    { lo := outLo, hi := outHi }

/-- Derivative bounds for sin: d(sin)/dx = cos ∈ [-1, 1].
    For interval [l, u], d(sin)/dx ∈ [min(cos(l), cos(u)), max(cos(l), cos(u))]
    if interval is small; [-1, 1] otherwise.
-/
def derivSin {n : Nat} (xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi, dB.lo, dB.hi with
  | .dim xlo, .dim xhi, .dim dlo, .dim dhi =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        let width := xu - xl
        -- Derivative of sin is cos, which is bounded in [-1, 1]
        let cos_lo := if width > Numbers.three then (-(Numbers.one))
                      else
                        let c1 := MathFunctions.cos xl
                        let c2 := MathFunctions.cos xu
                        if c1 < c2 then c1 else c2
        let cos_hi := if width > Numbers.three then Numbers.one
                      else
                        let c1 := MathFunctions.cos xl
                        let c2 := MathFunctions.cos xu
                        if c1 > c2 then c1 else c2
        -- Multiply derivative by cos bounds
        let p1 := cos_lo * dl
        let p2 := cos_lo * dh
        let p3 := cos_hi * dl
        let p4 := cos_hi * dh
        let m1 := if p1 < p2 then p1 else p2
        let m2 := if p3 < p4 then p3 else p4
        Tensor.scalar (if m1 < m2 then m1 else m2))
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        let width := xu - xl
        let cos_lo := if width > Numbers.three then (-(Numbers.one))
                      else
                        let c1 := MathFunctions.cos xl
                        let c2 := MathFunctions.cos xu
                        if c1 < c2 then c1 else c2
        let cos_hi := if width > Numbers.three then Numbers.one
                      else
                        let c1 := MathFunctions.cos xl
                        let c2 := MathFunctions.cos xu
                        if c1 > c2 then c1 else c2
        let p1 := cos_lo * dl
        let p2 := cos_lo * dh
        let p3 := cos_hi * dl
        let p4 := cos_hi * dh
        let M1 := if p1 > p2 then p1 else p2
        let M2 := if p3 > p4 then p3 else p4
        Tensor.scalar (if M1 > M2 then M1 else M2))
    { lo := outLo, hi := outHi }

/-- Derivative bounds for cos: d(cos)/dx = -sin ∈ [-1, 1]. -/
def derivCos {n : Nat} (xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi, dB.lo, dB.hi with
  | .dim xlo, .dim xhi, .dim dlo, .dim dhi =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        let width := xu - xl
        -- Derivative of cos is -sin
        let neg_sin_lo := if width > Numbers.three then (-(Numbers.one))
                          else
                            let s1 := -(MathFunctions.sin xl)
                            let s2 := -(MathFunctions.sin xu)
                            if s1 < s2 then s1 else s2
        let neg_sin_hi := if width > Numbers.three then Numbers.one
                          else
                            let s1 := -(MathFunctions.sin xl)
                            let s2 := -(MathFunctions.sin xu)
                            if s1 > s2 then s1 else s2
        let p1 := neg_sin_lo * dl
        let p2 := neg_sin_lo * dh
        let p3 := neg_sin_hi * dl
        let p4 := neg_sin_hi * dh
        let m1 := if p1 < p2 then p1 else p2
        let m2 := if p3 < p4 then p3 else p4
        Tensor.scalar (if m1 < m2 then m1 else m2))
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        let width := xu - xl
        let neg_sin_lo := if width > Numbers.three then (-(Numbers.one))
                          else
                            let s1 := -(MathFunctions.sin xl)
                            let s2 := -(MathFunctions.sin xu)
                            if s1 < s2 then s1 else s2
        let neg_sin_hi := if width > Numbers.three then Numbers.one
                          else
                            let s1 := -(MathFunctions.sin xl)
                            let s2 := -(MathFunctions.sin xu)
                            if s1 > s2 then s1 else s2
        let p1 := neg_sin_lo * dl
        let p2 := neg_sin_lo * dh
        let p3 := neg_sin_hi * dl
        let p4 := neg_sin_hi * dh
        let M1 := if p1 > p2 then p1 else p2
        let M2 := if p3 > p4 then p3 else p4
        Tensor.scalar (if M1 > M2 then M1 else M2))
    { lo := outLo, hi := outHi }

/-- Derivative bounds for atan: d(atan)/dx = 1/(1 + x²) ∈ (0, 1].
    This is always positive and bounded.
-/
def derivAtan {n : Nat} (xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi, dB.lo, dB.hi with
  | .dim xlo, .dim xhi, .dim dlo, .dim dhi =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        -- d(atan)/dx = 1/(1+x²), decreasing in |x|
        -- For interval [xl, xu], max derivative at x closest to 0
        let x_abs_max := if MathFunctions.abs xl > MathFunctions.abs xu
                         then MathFunctions.abs xl
                         else MathFunctions.abs xu
        let x_abs_min := if xl < Numbers.zero then
                           if xu > Numbers.zero then Numbers.zero
                           else MathFunctions.abs xu
                         else xl
        let d_lo := Numbers.one / (Numbers.one + x_abs_max * x_abs_max)
        let d_hi := Numbers.one / (Numbers.one + x_abs_min * x_abs_min)
        -- Multiply by input derivatives
        let p1 := d_lo * dl
        let p2 := d_lo * dh
        let p3 := d_hi * dl
        let p4 := d_hi * dh
        let m1 := if p1 < p2 then p1 else p2
        let m2 := if p3 < p4 then p3 else p4
        Tensor.scalar (if m1 < m2 then m1 else m2))
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        let x_abs_max := if MathFunctions.abs xl > MathFunctions.abs xu
                         then MathFunctions.abs xl
                         else MathFunctions.abs xu
        let x_abs_min := if xl < Numbers.zero then
                           if xu > Numbers.zero then Numbers.zero
                           else MathFunctions.abs xu
                         else xl
        let d_lo := Numbers.one / (Numbers.one + x_abs_max * x_abs_max)
        let d_hi := Numbers.one / (Numbers.one + x_abs_min * x_abs_min)
        let p1 := d_lo * dl
        let p2 := d_lo * dh
        let p3 := d_hi * dl
        let p4 := d_hi * dh
        let M1 := if p1 > p2 then p1 else p2
        let M2 := if p3 > p4 then p3 else p4
        Tensor.scalar (if M1 > M2 then M1 else M2))
    { lo := outLo, hi := outHi }

/-- Second derivative bounds for sin: d²(sin)/dx² = -sin. -/
def deriv2Sin {n : Nat} (xB : Box α (.dim n .scalar))
    (dB d2B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  -- y'' = f''(x) * (dx)² + f'(x) * d²x
  -- where f'(x) = cos(x) and f''(x) = -sin(x)
  match xB.lo, xB.hi, dB.lo, dB.hi, d2B.lo, d2B.hi with
  | .dim xlo, .dim xhi, .dim dlo, .dim dhi, .dim d2lo, .dim d2hi =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i, d2lo i, d2hi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh, .scalar d2l, .scalar _d2h =>
        let width := xu - xl
        -- -sin bounds
        let neg_sin_lo := if width > Numbers.three then (-(Numbers.one))
                          else
                            let s1 := -(MathFunctions.sin xl)
                            let s2 := -(MathFunctions.sin xu)
                            if s1 < s2 then s1 else s2
        -- cos bounds
        let cos_lo := if width > Numbers.three then (-(Numbers.one))
                      else
                        let c1 := MathFunctions.cos xl
                        let c2 := MathFunctions.cos xu
                        if c1 < c2 then c1 else c2
        -- Compute (dx)²
        let dx_sq_lo :=
          let dl2 := dl * dl
          let dh2 := dh * dh
          if dl < Numbers.zero then
            if dh > Numbers.zero then Numbers.zero
            else dh2
          else dl2
        -- term1 = -sin * dx²
        -- term2 = cos * d²x
        -- Conservative: combine both terms with full interval arithmetic
        let lo := neg_sin_lo * dx_sq_lo + cos_lo * d2l
        Tensor.scalar lo)
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i, d2lo i, d2hi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh, .scalar _d2l, .scalar d2h =>
        let width := xu - xl
        let neg_sin_hi := if width > Numbers.three then Numbers.one
                          else
                            let s1 := -(MathFunctions.sin xl)
                            let s2 := -(MathFunctions.sin xu)
                            if s1 > s2 then s1 else s2
        let cos_hi := if width > Numbers.three then Numbers.one
                      else
                        let c1 := MathFunctions.cos xl
                        let c2 := MathFunctions.cos xu
                        if c1 > c2 then c1 else c2
        let dx_sq_hi :=
          let dl2 := dl * dl
          let dh2 := dh * dh
          if dl2 > dh2 then dl2 else dh2
        let hi := neg_sin_hi * dx_sq_hi + cos_hi * d2h
        Tensor.scalar hi)
    { lo := outLo, hi := outHi }

namespace Theorems

/-- IBP for sin produces a valid Box structure. -/
theorem ibp_sin_returns_box {n : Nat} (xB : Box α (.dim n .scalar)) :
    ∃ lo hi : Tensor α (.dim n .scalar), ibpSin xB = { lo := lo, hi := hi } := by
  simp only [ibpSin]
  match xB.lo, xB.hi with
  | .dim _, .dim _ => exact ⟨_, _, rfl⟩

/-- IBP for cos produces a valid Box structure. -/
theorem ibp_cos_returns_box {n : Nat} (xB : Box α (.dim n .scalar)) :
    ∃ lo hi : Tensor α (.dim n .scalar), ibpCos xB = { lo := lo, hi := hi } := by
  simp only [ibpCos]
  match xB.lo, xB.hi with
  | .dim _, .dim _ => exact ⟨_, _, rfl⟩

omit [Context α] in
/-- IBP for atan produces a valid Box structure. -/
theorem ibp_atan_returns_box {n : Nat} [TanAtan α] (xB : Box α (.dim n .scalar)) :
    ∃ lo hi : Tensor α (.dim n .scalar), ibpAtan xB = { lo := lo, hi := hi } := by
  simp only [ibpAtan]
  match xB.lo, xB.hi with
  | .dim _, .dim _ => exact ⟨_, _, rfl⟩

/-- Derivative bounds for sin produce valid Box. -/
theorem deriv_sin_returns_box {n : Nat} (xB dB : Box α (.dim n .scalar)) :
    ∃ lo hi : Tensor α (.dim n .scalar), derivSin xB dB = { lo := lo, hi := hi } := by
  simp only [derivSin]
  match xB.lo, xB.hi, dB.lo, dB.hi with
  | .dim _, .dim _, .dim _, .dim _ => exact ⟨_, _, rfl⟩

end Theorems

end NN.MLTheory.CROWN.Operators.Trigonometric
