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

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ### Negation -/

/-- Negation, $f(x)=-x$, is the simplest linear operation. -/
def neg (x : α) : α := -x

/-- IBP for negation. Just swaps and negates bounds. -/
def ibpNegScalar (l u : α) : α × α :=
  (-u, -l)

/-- IBP for negation on boxes. -/
def ibpNeg (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim (fun i => Tensor.scalar (-B.hi.getScalar i))
  let outHi := Tensor.dim (fun i => Tensor.scalar (-B.lo.getScalar i))
  { lo := outLo, hi := outHi }

/-- Affine bounds for negation (exact). -/
def affNeg : α × α × α × α :=
  (-1, 0, -1, 0)

/-- Derivative of negation (constant -1). -/
def derivNeg : α × α := (-1, -1)

/-! ### Absolute Value -/

/-- Absolute value: $f(x)=|x|$. -/
def abs (x : α) : α :=
  if x > 0 then x else -x

/-- Interval propagation rule for scalar absolute value over `[l,u]`. -/
def ibpAbsScalar (l u : α) : α × α :=
  if l > 0 then
    -- All positive: |x| = x
    (l, u)
  else if u < 0 then
    -- All negative: |x| = -x
    (-u, -l)
  else
    -- Spans zero: min is 0, max is max(|l|, u)
    let absL := -l
    (0, if absL > u then absL else u)

/-- Apply the scalar absolute-value interval rule coordinatewise to a vector box. -/
def ibpAbs (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim (fun i =>
    Tensor.scalar (ibpAbsScalar (B.lo.getScalar i) (B.hi.getScalar i)).1)
  let outHi := Tensor.dim (fun i =>
    Tensor.scalar (ibpAbsScalar (B.lo.getScalar i) (B.hi.getScalar i)).2)
  { lo := outLo, hi := outHi }

/--
Affine lower and upper bounds for absolute value on an ordered interval `[l, u]`.

On an interval crossing zero, the lower bound is the zero line and the upper bound is the secant
through `(l, -l)` and `(u, u)`. The degenerate interval `[0, 0]` is represented by the zero line.
-/
def affAbs (l u : α) : α × α × α × α :=
  if l > 0 then
    (1, 0, 1, 0)
  else if u < 0 then
    (-1, 0, -1, 0)
  else
    if u > l then
      let slope := (u + l) / (u - l)
      let bias := u - slope * u
      (0, 0, slope, bias)
    else
      (0, 0, 0, 0)

/-! ### Reciprocal -/

/-- Reciprocal: $f(x)=1/x$. -/
def reciprocal (x : α) : α := 1 / x

/-- IBP for reciprocal on boxes, defined only when every coordinate interval excludes zero. -/
def ibpReciprocal? (n : Nat) (B : Box α (.dim n .scalar)) :
    Option (Box α (.dim n .scalar)) :=
  if (List.finRange n).all (fun i =>
      B.lo.getScalar i > 0 || B.hi.getScalar i < 0) then
    let outLo := Tensor.dim (fun i =>
      Tensor.scalar (1 / B.hi.getScalar i))
    let outHi := Tensor.dim (fun i =>
      Tensor.scalar (1 / B.lo.getScalar i))
    some { lo := outLo, hi := outHi }
  else
    none

/-! ### Power -/

/-- Helper for positive integer power. -/
def posPow (base : α) (exp : Nat) : α :=
  match exp with
  | 0 => 1
  | k + 1 => base * posPow base k

/-- Integer power: $f(x)=x^n$. -/
def powerInt (x : α) (n : Int) : α :=
  if n == 0 then 1
  else if n > 0 then
    posPow x n.toNat
  else
    -- Negative power: 1/x^|n|
    1 / posPow x (-n).toNat

/-- IBP for x². -/
def ibpSquareScalar (l u : α) : α × α :=
  let l2 := l * l
  let u2 := u * u
  if l > 0 then
    (l2, u2)
  else if u < 0 then
    (u2, l2)
  else
    -- Spans zero
    (0, if l2 > u2 then l2 else u2)

/-- IBP for x² on boxes. -/
def ibpSquare (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim (fun i =>
    Tensor.scalar (ibpSquareScalar (B.lo.getScalar i) (B.hi.getScalar i)).1)
  let outHi := Tensor.dim (fun i =>
    Tensor.scalar (ibpSquareScalar (B.lo.getScalar i) (B.hi.getScalar i)).2)
  { lo := outLo, hi := outHi }

/-- Affine bounds for x². -/
def affSquare (l u : α) : α × α × α × α :=
  -- x² is convex, so secant for upper, tangent for lower
  let slopeSec := l + u
  let biasSec := -(l * u)  -- Secant: y = (l+u)x - lu
  -- Tangent at midpoint
  let mid := (l + u) * (1 / 2)
  let slopeTan := 2 * mid
  let biasTan := -(mid * mid)
  (slopeTan, biasTan, slopeSec, biasSec)

/-! ### Min/Max -/

/-- Elementwise minimum of two boxes. -/
def ibpMin (n : Nat) (B1 B2 : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim (fun i =>
    let l1 := B1.lo.getScalar i
    let l2 := B2.lo.getScalar i
    Tensor.scalar (if l1 < l2 then l1 else l2))
  let outHi := Tensor.dim (fun i =>
    let u1 := B1.hi.getScalar i
    let u2 := B2.hi.getScalar i
    Tensor.scalar (if u1 < u2 then u1 else u2))
  { lo := outLo, hi := outHi }

/-- Elementwise maximum of two boxes. -/
def ibpMax (n : Nat) (B1 B2 : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim (fun i =>
    let l1 := B1.lo.getScalar i
    let l2 := B2.lo.getScalar i
    Tensor.scalar (if l1 > l2 then l1 else l2))
  let outHi := Tensor.dim (fun i =>
    let u1 := B1.hi.getScalar i
    let u2 := B2.hi.getScalar i
    Tensor.scalar (if u1 > u2 then u1 else u2))
  { lo := outLo, hi := outHi }

/--
Clamp one scalar with the same composition used by `Spec.clampSpec`:
`min clampHi (max clampLo x)`.

In particular, when `clampLo > clampHi`, the result is `clampHi`. This agrees with PyTorch's
documented behavior instead of silently switching the two bounds.
-/
def clampScalar (x clampLo clampHi : α) : α :=
  let floored := if x > clampLo then x else clampLo
  if floored < clampHi then floored else clampHi

/-- Clamp operation: `clamp(x, lo, hi) = min(hi, max(lo, x))`. -/
def ibpClampScalar (xLo xHi clampLo clampHi : α) : α × α :=
  (clampScalar xLo clampLo clampHi, clampScalar xHi clampLo clampHi)

/-- Interval propagation for `clamp`, applied coordinatewise to a vector box. -/
def ibpClamp (n : Nat) (B : Box α (.dim n .scalar)) (clampLo clampHi : α) : Box α (.dim n
  .scalar) :=
  let outLo := Tensor.dim (fun i =>
    Tensor.scalar (clampScalar (B.lo.getScalar i) clampLo clampHi))
  let outHi := Tensor.dim (fun i =>
    Tensor.scalar (clampScalar (B.hi.getScalar i) clampLo clampHi))
  { lo := outLo, hi := outHi }

end NN.MLTheory.CROWN.Operators.Arithmetic
