/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

/-
Arithmetic operators for CROWN bound propagation.

This file implements IBP and affine bounds for:
- Power: f(x) = x^n
- Sqrt: f(x) = √x
- Neg: f(x) = -x
- Reciprocal: f(x) = 1/x
- Abs: f(x) = |x|
- Min/Max: f(x,y) = min(x,y) / max(x,y)
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# `NN.MLTheory.CROWN.Operators.Arithmetic`

IBP and affine transfer rules for arithmetic primitives (negation, absolute value, reciprocal,
square root, powers, min/max) used by the CROWN bound propagation engine.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Arithmetic

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-! ### Negation -/

/-- Negation: f(x) = -x. Simplest linear operation. -/
def neg (x : α) : α := -x

/-- IBP for negation. Just swaps and negates bounds. -/
def ibpNegScalar (l u : α) : α × α :=
  (-u, -l)

/-- IBP for negation on boxes. -/
def ibpNeg (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (-u))
    let outHi := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (-l))
    { lo := outLo, hi := outHi }

/-- Affine bounds for negation (exact). -/
def affNeg : α × α × α × α :=
  (-Numbers.one, Numbers.zero, -Numbers.one, Numbers.zero)

/-- Derivative of negation (constant -1). -/
def derivNeg : α × α := (-Numbers.one, -Numbers.one)

/-! ### Absolute Value -/

/-- Absolute value: f(x) = |x|. -/
def abs (x : α) : α :=
  if x > Numbers.zero then x else -x

/-- IBP for absolute value. -/
def ibpAbsScalar (l u : α) : α × α :=
  if l > Numbers.zero then
    -- All positive: |x| = x
    (l, u)
  else if u < Numbers.zero then
    -- All negative: |x| = -x
    (-u, -l)
  else
    -- Spans zero: min is 0, max is max(|l|, u)
    let absL := -l
    (Numbers.zero, if absL > u then absL else u)

/-- IBP for absolute value on boxes. -/
def ibpAbs (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpAbsScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpAbsScalar l u).2)
    { lo := outLo, hi := outHi }

/-- Affine bounds for absolute value. -/
def affAbs (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if u < Numbers.zero then
    (-Numbers.one, Numbers.zero, -Numbers.one, Numbers.zero)
  else
    -- Crossing: upper bound is secant, lower is 0
    let slope := ((-l) + u) / (u - l)  -- This simplifies but keeping for clarity
    (Numbers.zero, Numbers.zero, slope, -slope * l)

/-! ### Square Root -/

/-- Square-root approximation using two Babylonian refinement steps for a `Context` scalar. -/
def sqrtApprox (x : α) : α :=
  if x < Numbers.epsilon then
    Numbers.zero
  else
    -- Babylonian method with initial guess
    let guess := (x + Numbers.one) * Numbers.pointfive
    let refined := (guess + x / guess) * Numbers.pointfive
    (refined + x / refined) * Numbers.pointfive

/-- IBP for square root. Sqrt is monotone increasing on [0,∞). -/
def ibpSqrtScalar (l u : α) : α × α :=
  -- Assuming l ≥ 0 (sqrt domain)
  let l' := if l < Numbers.zero then Numbers.zero else l
  (sqrtApprox l', sqrtApprox u)

/-- IBP for square root on boxes. -/
def ibpSqrt (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (ibpSqrtScalar l Numbers.zero).1)
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (ibpSqrtScalar Numbers.zero u).2)
    { lo := outLo, hi := outHi }

/-- Affine bounds for square root (concave function).
    Upper: tangent line, Lower: secant line. -/
def affSqrt (l u : α) : α × α × α × α :=
  let sl := sqrtApprox l
  let su := sqrtApprox u
  -- Secant for lower bound (since sqrt is concave)
  let slope_sec := if u > l + Numbers.epsilon then (su - sl) / (u - l) else Numbers.one /
    (Numbers.two * sl)
  let bias_sec := sl - slope_sec * l
  -- Tangent at midpoint for upper bound
  let mid := (l + u) * Numbers.pointfive
  let smid := sqrtApprox mid
  let slope_tan := if smid > Numbers.epsilon then Numbers.one / (Numbers.two * smid) else
    Numbers.one
  let bias_tan := smid - slope_tan * mid
  (slope_sec, bias_sec, slope_tan, bias_tan)

/-- Derivative bounds for sqrt: d/dx √x = 1/(2√x). -/
def derivSqrt (l u : α) : α × α :=
  let sl := sqrtApprox l
  let su := sqrtApprox u
  -- Derivative decreases as x increases
  let deriv_lo := if su > Numbers.epsilon then Numbers.one / (Numbers.two * su) else Numbers.one
  let deriv_hi := if sl > Numbers.epsilon then Numbers.one / (Numbers.two * sl) else Numbers.one
  (deriv_lo, deriv_hi)

/-! ### Reciprocal -/

/-- Reciprocal: f(x) = 1/x. -/
def reciprocal (x : α) : α := Numbers.one / x

/-- IBP for reciprocal.
    Warning: 1/x has asymptote at 0, so this is only valid when 0 ∉ [l,u]. -/
def ibpReciprocalScalar (l u : α) : α × α :=
  if l > Numbers.zero then
    -- All positive: 1/x is decreasing
    (Numbers.one / u, Numbers.one / l)
  else if u < Numbers.zero then
    -- All negative: 1/x is decreasing
    (Numbers.one / u, Numbers.one / l)
  else
    -- Contains zero: return very wide bounds
    let inf := Numbers.one / Numbers.epsilon
    (-inf, inf)

/-- IBP for reciprocal on boxes. -/
def ibpReciprocal (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpReciprocalScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpReciprocalScalar l u).2)
    { lo := outLo, hi := outHi }

/-- Affine bounds for reciprocal (convex for x > 0).
    Lower: secant line, Upper: tangent line. -/
def affReciprocal (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    let rl := Numbers.one / l
    let ru := Numbers.one / u
    -- Secant for lower (since 1/x is convex on x > 0)
    let slope_sec := (ru - rl) / (u - l)
    let bias_sec := rl - slope_sec * l
    -- Tangent at midpoint for upper
    let mid := (l + u) * Numbers.pointfive
    let rmid := Numbers.one / mid
    let slope_tan := -(Numbers.one / (mid * mid))
    let bias_tan := rmid - slope_tan * mid
    (slope_sec, bias_sec, slope_tan, bias_tan)
  else
    -- Contains zero or negative: very conservative
    let inf := Numbers.one / Numbers.epsilon
    (Numbers.zero, -inf, Numbers.zero, inf)

/-- Derivative bounds for reciprocal: d/dx (1/x) = -1/x². -/
def derivReciprocal (l u : α) : α × α :=
  if l > Numbers.zero then
    -- All positive: -1/x² is negative, |deriv| decreases as x increases
    let deriv_lo := -(Numbers.one / (l * l))
    let deriv_hi := -(Numbers.one / (u * u))
    (deriv_hi, deriv_lo)  -- Note: deriv_hi > deriv_lo since both negative
  else
    let inf := Numbers.one / Numbers.epsilon
    (-inf, Numbers.zero)

/-! ### Power -/

/-- Helper for positive integer power. -/
def posPow (base : α) (exp : Nat) : α :=
  match exp with
  | 0 => Numbers.one
  | k + 1 => base * posPow base k

/-- Power: f(x) = x^n (integer power). -/
def powerInt (x : α) (n : Int) : α :=
  if n == 0 then Numbers.one
  else if n > 0 then
    posPow x n.toNat
  else
    -- Negative power: 1/x^|n|
    Numbers.one / posPow x (-n).toNat

/-- IBP for x². -/
def ibpSquareScalar (l u : α) : α × α :=
  let l2 := l * l
  let u2 := u * u
  if l > Numbers.zero then
    (l2, u2)
  else if u < Numbers.zero then
    (u2, l2)
  else
    -- Spans zero
    (Numbers.zero, if l2 > u2 then l2 else u2)

/-- IBP for x² on boxes. -/
def ibpSquare (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSquareScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSquareScalar l u).2)
    { lo := outLo, hi := outHi }

/-- Affine bounds for x². -/
def affSquare (l u : α) : α × α × α × α :=
  -- x² is convex, so secant for upper, tangent for lower
  let slope_sec := l + u
  let bias_sec := -(l * u)  -- Secant: y = (l+u)x - lu
  -- Tangent at midpoint
  let mid := (l + u) * Numbers.pointfive
  let slope_tan := Numbers.two * mid
  let bias_tan := -(mid * mid)
  (slope_tan, bias_tan, slope_sec, bias_sec)

/-! ### Min/Max -/

/-- Elementwise minimum of two boxes. -/
def ibpMin (n : Nat) (B1 B2 : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B1.lo, B1.hi, B2.lo, B2.hi with
  | .dim lo1, .dim hi1, .dim lo2, .dim hi2 =>
    let outLo := Tensor.dim (fun i =>
      match lo1 i, lo2 i with
      | .scalar l1, .scalar l2 =>
        Tensor.scalar (if l1 < l2 then l1 else l2))
    let outHi := Tensor.dim (fun i =>
      match hi1 i, hi2 i with
      | .scalar u1, .scalar u2 =>
        -- max is min of upper bounds
        Tensor.scalar (if u1 < u2 then u1 else u2))
    { lo := outLo, hi := outHi }

/-- Elementwise maximum of two boxes. -/
def ibpMax (n : Nat) (B1 B2 : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B1.lo, B1.hi, B2.lo, B2.hi with
  | .dim lo1, .dim hi1, .dim lo2, .dim hi2 =>
    let outLo := Tensor.dim (fun i =>
      match lo1 i, lo2 i with
      | .scalar l1, .scalar l2 =>
        -- min is max of lower bounds
        Tensor.scalar (if l1 > l2 then l1 else l2))
    let outHi := Tensor.dim (fun i =>
      match hi1 i, hi2 i with
      | .scalar u1, .scalar u2 =>
        Tensor.scalar (if u1 > u2 then u1 else u2))
    { lo := outLo, hi := outHi }

/-- Clamp operation: clamp(x, lo, hi) = max(lo, min(hi, x)). -/
def ibpClampScalar (x_lo x_hi clamp_lo clamp_hi : α) : α × α :=
  -- Output is in [clamp_lo, clamp_hi] intersected with [x_lo, x_hi]
  let out_lo := if x_lo > clamp_lo then x_lo else clamp_lo
  let out_hi := if x_hi < clamp_hi then x_hi else clamp_hi
  (out_lo, out_hi)

/-- IBP for clamp. -/
def ibpClamp (n : Nat) (B : Box α (.dim n .scalar)) (clamp_lo clamp_hi : α) : Box α (.dim n
  .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l =>
        Tensor.scalar (if l > clamp_lo then l else clamp_lo))
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u =>
        Tensor.scalar (if u < clamp_hi then u else clamp_hi))
    { lo := outLo, hi := outHi }

namespace Theorems

/-- Negation IBP is exact. This states the basic structure of negation bounds. -/
theorem neg_ibp_scalar_structure (l u : α) :
    ibpNegScalar (α:=α) l u = (-u, -l) := by
  rfl

/-- Absolute value IBP returns a pair. -/
theorem abs_ibp_returns_pair (l u : α) :
    ∃ lo hi : α, ibpAbsScalar (α:=α) l u = (lo, hi) := by
  exact ⟨(ibpAbsScalar l u).1, (ibpAbsScalar l u).2, rfl⟩

/-- Square IBP returns a pair. -/
theorem square_ibp_returns_pair (l u : α) :
    ∃ lo hi : α, ibpSquareScalar (α:=α) l u = (lo, hi) := by
  exact ⟨(ibpSquareScalar l u).1, (ibpSquareScalar l u).2, rfl⟩

end Theorems

end NN.MLTheory.CROWN.Operators.Arithmetic
